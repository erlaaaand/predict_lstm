import yfinance as yf
import pandas as pd
from datetime import date, timedelta

TICKERS = [
    "MBMA.JK"
]

# Mengambil data lebih banyak untuk memastikan mendapat 90 hari perdagangan
# Karena ada hari libur/weekend, kita ambil data ~6 bulan ke belakang
today = date.today()
end_date = today + timedelta(days=1)
start_date = today - timedelta(days=180)  # Ambil 180 hari untuk memastikan dapat 90 hari perdagangan

print(f"Mengambil 90 hari perdagangan (Open, High, Low, Close & Volume) untuk ticker: {', '.join(TICKERS)}")
print(f"Periode pencarian: {start_date} hingga {today}")
print("(Hari libur dan weekend akan otomatis dilewati)\n")

try:
    # Mengambil data harian dengan auto_adjust=False untuk menghindari warning
    stock_data = yf.download(TICKERS, start=start_date, end=end_date, auto_adjust=False)

    if stock_data.empty:
        print("Tidak ada data yang ditemukan. Periksa kembali ticker saham atau periode waktu.")
    else:
        # Jika hanya satu ticker, flatten column MultiIndex
        if len(TICKERS) == 1:
            stock_data.columns = stock_data.columns.get_level_values(0)
        
        # Ambil hanya 90 data perdagangan terakhir
        stock_data = stock_data.tail(90)
        
        # Memperbaiki data Volume yang kosong (NaN) menjadi 0
        stock_data['Volume'] = stock_data['Volume'].fillna(0).astype(int)

        # Menghapus komponen waktu dari indeks tanggal agar rapi
        stock_data.index = stock_data.index.date

        # Membuat DataFrame gabungan untuk tampilan yang lebih rapi
        summary_df = pd.DataFrame({
            'Harga Open': stock_data['Open'],
            'Harga High': stock_data['High'],
            'Harga Low': stock_data['Low'],
            'Harga Close': stock_data['Close'],
            'Volume': stock_data['Volume']
        })

        # Menampilkan informasi jumlah data
        total_data = len(summary_df)
        tanggal_awal = summary_df.index[0]
        tanggal_akhir = summary_df.index[-1]
        
        print(f"✓ Berhasil mengambil {total_data} hari perdagangan")
        print(f"  Dari tanggal: {tanggal_awal} hingga {tanggal_akhir}\n")

        # Menampilkan data terbaru (10 hari terakhir)
        print("=" * 90)
        print("DATA HARGA SAHAM HARIAN (10 Hari Perdagangan Terakhir)")
        print("=" * 90)
        print(summary_df.tail(10).to_string())
        print("=" * 90)

        # Memisahkan data untuk keperluan lain
        open_prices = stock_data['Open']
        high_prices = stock_data['High']
        low_prices = stock_data['Low']
        close_prices = stock_data['Close']
        volumes = stock_data['Volume']

        # Statistik ringkasan
        print(f"\n--- STATISTIK RINGKASAN ({total_data} Hari Perdagangan) ---")
        print(f"Open  - Min: {open_prices.min():.2f}, Max: {open_prices.max():.2f}, Rata-rata: {open_prices.mean():.2f}")
        print(f"High  - Min: {high_prices.min():.2f}, Max: {high_prices.max():.2f}, Rata-rata: {high_prices.mean():.2f}")
        print(f"Low   - Min: {low_prices.min():.2f}, Max: {low_prices.max():.2f}, Rata-rata: {low_prices.mean():.2f}")
        print(f"Close - Min: {close_prices.min():.2f}, Max: {close_prices.max():.2f}, Rata-rata: {close_prices.mean():.2f}")
        print(f"Volume - Min: {volumes.min():,}, Max: {volumes.max():,}, Rata-rata: {volumes.mean():,.0f}")

        # Menyimpan setiap tabel ke dalam sheet yang berbeda di satu file Excel
        excel_file_name = 'harga_saham_90_hari_perdagangan.xlsx'
        with pd.ExcelWriter(excel_file_name, engine='openpyxl') as writer:
            # Sheet ringkasan dengan Open, High, Low, Close, dan Volume
            summary_df.to_excel(writer, sheet_name='Ringkasan')
            
            # Sheet terpisah untuk setiap data
            open_prices.to_frame(name='Harga Open').to_excel(writer, sheet_name='Harga Open')
            high_prices.to_frame(name='Harga High').to_excel(writer, sheet_name='Harga High')
            low_prices.to_frame(name='Harga Low').to_excel(writer, sheet_name='Harga Low')
            close_prices.to_frame(name='Harga Close').to_excel(writer, sheet_name='Harga Close')
            volumes.to_frame(name='Volume').to_excel(writer, sheet_name='Volume Perdagangan')
            
            # Data lengkap dengan semua kolom
            stock_data.to_excel(writer, sheet_name='Data Lengkap Harian')

        print(f"\n✓ Data berhasil disimpan ke file Excel '{excel_file_name}'")
        print(f"  Total: {total_data} hari perdagangan (hari libur/weekend otomatis dilewati)")
        print(f"  Sheet tersedia: Ringkasan, Harga Open, High, Low, Close, Volume, Data Lengkap")

except Exception as e:
    print(f"✗ Terjadi kesalahan saat mengambil data: {e}")